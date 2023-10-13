#include "gmm_shm_nv.h"

// alloc nv gpu mem
// out: ret_ptr, shared_fd
CUresult gmm_nv_vmm_alloc(Cudeviceptr &ret_ptr, size_t size, int dev_id,
                          size_t vmm_granularity, int &shared_fd,
                          bool export_flag) {
  CUresult ret = CUDA_SUCCESS;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmemGenericAllocationHandle vmm_handle;
  ret = __CF("cuMemCreate")(&vmm_handle, size, &prop, 0);

  size_t aligned_size = alignSize(size, vmm_granularity);

  CUmemAccessDesc accessDesc;
  accessDesc.location = dev_id;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  ret = __CF("cuMemAddressReserve")(&ret_ptr, aligned_size, 0ULL, 0U, 0);
  ret = __CF("cuMemMap")((CUdeviceptr)ret_ptr, aligned_size, 0ULL, vmm_handle,
                         0ULL);
  ret = __CF("cuMemSetAccess")((CUdeviceptr)ret_ptr, aligned_size, accessDescs,
                               1);
  ret = __CF("cuMemExportToShareableHandle")(
      (void *)&shared_fd, vmm_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      0);

  return ret;
}
