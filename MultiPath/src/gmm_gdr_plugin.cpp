#include "gmm_gdr_plugin.h"

#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <map>

#include "gmm_api_stats.h"
#include "gmm_cuda_mem.h"
#include "gmm_util.h"

using namespace std;
#define MODULE_STATUS CUresult
// notice: use __CF("") to call CUDA directly, avoiding invoke hook again
extern void *libP;

// check whether the GPU support gdr or not
// true: support
// false: not support
bool check_gdr_support(CUdevice dev) {
#if CUDA_VERSION >= 11030
  int drv_version;
  ASSERTDRV(cuDriverGetVersion(&drv_version));

  // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA
  // support.
  if (drv_version >= 11030) {
    int gdr_support = 0;
    ASSERTDRV(cuDeviceGetAttribute(
        &gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));

    if (!gdr_support) printf("This GPU does not support GPUDirect RDMA.\n");

    return !!gdr_support;
  }
#endif

  // For older versions, we fall back to detect this support with
  // gdr_pin_buffer.
  const size_t size = GPU_PAGE_SIZE;
  CUdeviceptr d_A;

  gdr_t gH = gdr_open_safe();
  if (!gH) {
    return false;
  }

  int ret = cuMemAlloc(&d_A, size);
  if (ret != CUDA_SUCCESS) return false;

  gdr_mh_t mH;

  ret = gdr_pin_buffer(gH, d_A, size, 0, 0, &mH);
  if (ret != 0) {
    printf("error in gdr_pin_buffer with code=%d\n", ret);
    printf("Your GPU might not support GPUDirect RDMA\n");
  } else
    ASSERT_EQ(gdr_unpin_buffer(gH, mH), 0);

  ASSERT_EQ(gdr_close(gH), 0);
  cuMemFree(d_A);

  return ret == 0;
}

// check whether /dev/gdrdrv is ready and gdrdrv.ko is loaded
static const char gdr_device[] = "/dev/gdrdrv";
bool check_gdrdrv_ready() {
  bool ret = gmm_is_file_exist(gdr_device);
  if (!ret) {
    printf(
        "GDR is not ready, %s doesn't exist, ensure insmod gdrdrv, and use "
        "--device=/dev/gdrdrv if run with container\n",
        gdr_device);
  }
  return ret;
}

// map the CUDA dev mem to VA
// pre: dev mem dptr already allocated
// input: gH, dptr, in_size
// output: [mH, map_dptr, (aligned)va_dptr(for IO), aligned_size]
int gmm_gdr_map(gdr_t &gH, CUdeviceptr &dptr, size_t in_size, gdr_mh_t &mH,
                gdr_info_t &info, void *&va_dptr, void *&map_dptr,
                size_t &aligned_size) {
  int ret = 0;

  aligned_size = PAGE_ROUND_UP(in_size, GPU_PAGE_SIZE);

  ASSERT_EQ(gdr_pin_buffer(gH, dptr, aligned_size, 0, 0, &mH), 0);

  ASSERT_EQ(gdr_map(gH, mH, &map_dptr, aligned_size), 0);

  ASSERT_EQ(gdr_get_info(gH, mH, &info), 0);

  if (0) {
    std::cout << "info.va: " << std::hex << info.va << std::dec << std::endl;
    std::cout << "info.mapped_size: " << info.mapped_size << std::endl;
    std::cout << "info.page_size: " << info.page_size << std::endl;
    std::cout << "info.mapped: " << info.mapped << std::endl;
    std::cout << "info.wc_mapping: " << info.wc_mapping << std::endl;
  }

  int off = info.va - dptr;
  va_dptr = (uint32_t *)((char *)map_dptr + off);
  // cout << "user-space pointer: " << buf_ptr << endl;

  return ret;
}

// unmap and unpin map_addr and mHandle
void gmm_gdr_unmap(gdr_t &gH, gdr_mh_t &mH, void *&map_dptr,
                   size_t aligned_size) {
  ASSERT_EQ(gdr_unmap(gH, mH, map_dptr, aligned_size), 0);
  ASSERT_EQ(gdr_unpin_buffer(gH, mH), 0);
}
