#include <cuda.h>
#include <unistd.h>

#include <cstring>

#include "gmm_api_stats.h"
#include "gmm_cuda_mem.h"
#include "gmm_gdr_plugin.h"
#include "gmm_util.h"

#define MODULE_STATUS CUresult
// notice: use __CF("") to call CUDA directly, avoiding invoke hook again
extern void *libP;

CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size,
                       bool aligned_mapping, bool set_sync_memops) {
  CUresult ret = CUDA_SUCCESS;
  CUdeviceptr ptr, out_ptr;
  size_t allocated_size;

  if (aligned_mapping)
    allocated_size = PAGE_ROUND_UP(size, GPU_PAGE_SIZE);
  else
    allocated_size = size;

  ret = cuMemAlloc(&ptr, allocated_size);
  if (ret != CUDA_SUCCESS) return ret;

  if (set_sync_memops) {
    unsigned int flag = 1;
    ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
    if (ret != CUDA_SUCCESS) {
      cuMemFree(ptr);
      return ret;
    }
  }

  if (aligned_mapping)
    out_ptr = PAGE_ROUND_UP(ptr, GPU_PAGE_SIZE);
  else
    out_ptr = ptr;

  handle->ptr = out_ptr;
  handle->unaligned_ptr = ptr;
  handle->size = size;
  handle->allocated_size = allocated_size;

  return CUDA_SUCCESS;
}

CUresult gpu_mem_free(gpu_mem_handle_t *handle) {
  CUresult ret = CUDA_SUCCESS;
  CUdeviceptr ptr;

  ret = cuMemFree(handle->unaligned_ptr);
  if (ret == CUDA_SUCCESS) memset(handle, 0, sizeof(gpu_mem_handle_t));

  return ret;
}

#if CUDA_VERSION >= 11000
/**
 * Allocating GPU memory using VMM API.
 * VMM API is available since CUDA 10.2. However, the RDMA support is added in
 * CUDA 11.0. Our tests are not useful without RDMA support. So, we enable this
 * VMM allocation from CUDA 11.0.
 */
CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size,
                       bool aligned_mapping, bool set_sync_memops) {
  CUresult ret = CUDA_SUCCESS;

  size_t granularity, gran;
  CUmemAllocationProp mprop;
  CUdevice gpu_dev;
  size_t rounded_size;
  CUdeviceptr ptr = 0;
  CUmemGenericAllocationHandle mem_handle = 0;

  bool is_mapped = false;
  int RDMASupported = 0;
  int version;

  ret = cuDriverGetVersion(&version);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuDriverGetVersion\n");
    goto out;
  }

  if (version < 11000) {
    printf("VMM with RDMA is not supported in this CUDA version.\n");
    ret = CUDA_ERROR_NOT_SUPPORTED;
    goto out;
  }

  ret = cuCtxGetDevice(&gpu_dev);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuCtxGetDevice\n");
    goto out;
  }

  ret = cuDeviceGetAttribute(
      &RDMASupported,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, gpu_dev);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuDeviceGetAttribute\n");
    goto out;
  }

  if (!RDMASupported) {
    printf("GPUDirect RDMA is not supported on this GPU.\n");
    ret = CUDA_ERROR_NOT_SUPPORTED;
    goto out;
  }

  memset(&mprop, 0, sizeof(CUmemAllocationProp));
  mprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  mprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mprop.location.id = gpu_dev;
  mprop.allocFlags.gpuDirectRDMACapable = 1;

  ret = cuMemGetAllocationGranularity(&gran, &mprop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemGetAllocationGranularity\n");
    goto out;
  }

  // In case gran is smaller than GPU_PAGE_SIZE
  // alloc at least 2MB, may issue IO at any offset
  granularity = PAGE_ROUND_UP(gran, GPU_PAGE_SIZE);

  rounded_size = PAGE_ROUND_UP(size, granularity);
  ret = cuMemAddressReserve(&ptr, rounded_size, granularity, 0, 0);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemAddressReserve\n");
    goto out;
  }

  ret = cuMemCreate(&mem_handle, rounded_size, &mprop, 0);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemCreate\n");
    goto out;
  }

  ret = cuMemMap(ptr, rounded_size, 0, mem_handle, 0);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemMap\n");
    goto out;
  }
  is_mapped = true;

  CUmemAccessDesc access;
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = gpu_dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  ret = cuMemSetAccess(ptr, rounded_size, &access, 1);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemSetAccess\n");
    goto out;
  }

  // cuMemAddressReserve always returns aligned ptr
  handle->ptr = ptr;
  handle->handle = mem_handle;
  handle->size = size;
  handle->allocated_size = rounded_size;

out:
  if (ret != CUDA_SUCCESS) {
    if (is_mapped) cuMemUnmap(ptr, rounded_size);

    if (mem_handle) cuMemRelease(mem_handle);

    if (ptr) cuMemAddressFree(ptr, rounded_size);
  }
  return ret;
}

CUresult gpu_vmm_free(gpu_mem_handle_t *handle) {
  CUresult ret;

  if (!handle || !handle->ptr) return CUDA_ERROR_INVALID_VALUE;

  ret = cuMemUnmap(handle->ptr, handle->allocated_size);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemUnmap\n");
    return ret;
  }

  ret = cuMemRelease(handle->handle);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemRelease\n");
    return ret;
  }

  ret = cuMemAddressFree(handle->ptr, handle->allocated_size);
  if (ret != CUDA_SUCCESS) {
    printf("error in cuMemAddressFree\n");
    return ret;
  }

  memset(handle, 0, sizeof(gpu_mem_handle_t));

  return CUDA_SUCCESS;
}
#else
/* VMM with RDMA is not available before CUDA 11.0 */
CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size,
                       bool aligned_mapping, bool set_sync_memops) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult gpu_vmm_free(gpu_mem_handle_t *handle) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif

// alloc dev mem via VMM and export it
// assuming a mem pool thus limited dev mem allocation
CUresult gmm_cuda_vmm_alloc(size_t bytesize, int dev_id, CUdeviceptr &ret_ptr,
                            CUmemGenericAllocationHandle &vmm_handle,
                            size_t *alloc_size, bool enable_gdr, bool rw) {
  CUresult ret = CUDA_SUCCESS;

  size_t aligned_size = alignSize(bytesize, 2UL << 20);
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  if (enable_gdr) prop.allocFlags.gpuDirectRDMACapable = 1;

  ret = __CF("cuMemCreate")(&vmm_handle, aligned_size, &prop, 0);
  if (ret != 0) {
    goto FAST_RETURN;
  }
  ret = __CF("cuMemAddressReserve")(&ret_ptr, aligned_size, 0ULL, 0U, 0);
  if (ret != 0) {
    goto FAST_RETURN;
  }
  ret = __CF("cuMemMap")(ret_ptr, aligned_size, 0ULL, vmm_handle, 0ULL);
  if (ret != 0) {
  FAST_RETURN:
    printf("Error: cuMemMap ret=%d\n", ret);
    return ret;
  }

  CUmemAccessDesc accessDesc;
  prop.location.id = dev_id;
  accessDesc.location = prop.location;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!rw) accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  *alloc_size = aligned_size;
  return __CF("cuMemSetAccess")(ret_ptr, aligned_size, &accessDesc, 1);
}

CUresult gmm_cuda_vmm_free(CUdeviceptr &dptr, gmm_shmInfo_t *&shm) {
  CUresult ret = CUDA_SUCCESS;
  ret = __CF("cuMemUnmap")(dptr, shm->get_size());
  ret = __CF("cuMemAddressFree")(dptr, shm->get_size());
  ret = __CF("cuMemRelease")(shm->get_handle());

  // TODO: clean up other info e.g. shmInfo, small
  return ret;
}
