#pragma once

#include <stdio.h>
#include <cuda.h>
#include <gdrapi.h>
#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <map>

/**
 * Memory barrier
 */
#if defined(GDRAPI_X86)
#define MB() asm volatile("mfence" ::: "memory")
#elif defined(GDRAPI_POWER)
#define MB() asm volatile("sync" ::: "memory")
#else
#define MB() asm volatile("" ::: "memory")
#endif

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC
#define EXIT_WAIVED 2

#define ASSERT_IF(x)                                                      \
  do {                                                                    \
    if (!(x)) {                                                           \
      fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, \
              __LINE__);                                                  \
    }                                                                     \
  } while (0)

#define ASSERTDRV(stmt)                               \
  do {                                                \
    CUresult result = (stmt);                         \
    if (result != CUDA_SUCCESS) {                     \
      const char *_err_name;                          \
      cuGetErrorName(result, &_err_name);             \
      fprintf(stderr, "CUDA error: %s\n", _err_name); \
    }                                                 \
  } while (0)

#define ASSERT_EQ(P, V) ASSERT_IF((P) == (V))
#define CHECK_EQ(P, V) ASSERT_IF((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT_IF(!((P) == (V)))
#define BREAK_IF_NEQ(P, V) \
  if ((P) != (V)) break
#define BEGIN_CHECK do
#define END_CHECK while (0)

#define PAGE_ROUND_UP(x, n) (((x) + ((n)-1)) & ~((n)-1))

// check given GPU support gdr or not
// true: support
// false: not support
bool check_gdr_support(CUdevice dev);

// check whether /dev/gdrdrv file exist or not
bool check_gdrdrv_ready();

// return gH(ptr)
// nullptr: error
static inline gdr_t gdr_open_safe() {
  gdr_t gH = gdr_open();
  if (!gH) {
    fprintf(stderr, "gdr_open error: Is gdrdrv driver installed and loaded?\n");
  }
  return gH;
}

// init gdr if supported,
// true: support, and gH is valid
// false: not support, gH is nullptr
static bool gmm_gdr_open(gdr_t &gH) {
  bool ret = false;
  CUdevice dev;

  if (check_gdrdrv_ready() && (((gH = gdr_open_safe())) != nullptr) &&
      (cuCtxGetDevice(&dev) == CUDA_SUCCESS)) {
    ret = check_gdr_support(dev);
  }

  return ret;
}

// close gdr
static void gmm_gdr_close(gdr_t &gH) { ASSERT_EQ(gdr_close(gH), 0); }

// map the CUDA dev mem to VA
// pre: dev mem dptr already allocated
// input: gH, dptr, in_size
// output: [mH, map_dptr, (aligned)va_dptr(for IO), aligned_size]
int gmm_gdr_map(gdr_t &gH, CUdeviceptr &dptr, size_t in_size, gdr_mh_t &mH,
                gdr_info_t &info, void *&va_dptr, void *&map_dptr,
                size_t &aligned_size);

// unmap and unpin map_addr and mHandle
void gmm_gdr_unmap(gdr_t &gH, gdr_mh_t &mH, void *&map_dptr,
                   size_t aligned_size);

static inline int gmm_gdr_htod(gdr_mh_t &mH, void *aligned_va_dptr,
                               const void *host_pin_buf, size_t bytes) {
  return gdr_copy_to_mapping(mH, aligned_va_dptr, host_pin_buf, bytes);
  // MB();
}

static inline int gmm_gdr_dtoh(gdr_mh_t &mH, void *host_pin_buf,
                               const void *aligned_vm_dptr, size_t bytes) {
  return gdr_copy_from_mapping(mH, host_pin_buf, aligned_vm_dptr, bytes);
}
