#pragma once
// for dev mem management

#include <cuda.h>
#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <map>

#include "gdrapi.h"
#include "gmm_host_shm.h"

#define PAGE_ROUND_UP(x, n) (((x) + ((n)-1)) & ~((n)-1))

typedef struct gpuMemHandle {
  CUdeviceptr
      ptr;  // aligned ptr if requested; otherwise, the same as unaligned_ptr.
  union {
    CUdeviceptr unaligned_ptr;  // for tracking original ptr; may be unaligned.
#if CUDA_VERSION >= 11000
    // VMM with GDR support is available from CUDA 11.0
    CUmemGenericAllocationHandle handle;
#endif
  };
  size_t size;
  size_t allocated_size;
} gpu_mem_handle_t;

typedef CUresult (*gpu_memalloc_fn_t)(gpu_mem_handle_t *handle,
                                      const size_t size, bool aligned_mapping,
                                      bool set_sync_memops);
typedef CUresult (*gpu_memfree_fn_t)(gpu_mem_handle_t *handle);

CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size,
                       bool aligned_mapping, bool set_sync_memops);
CUresult gpu_mem_free(gpu_mem_handle_t *handle);

CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size,
                       bool aligned_mapping, bool set_sync_memops);
CUresult gpu_vmm_free(gpu_mem_handle_t *handle);

enum cuda_mem_type {
  GMM_MEM_TYPE_INVALID = 0,
  GMM_MEM_TYPE_UM = 1,
  GMM_MEM_TYPE_ZEROCOPY = 2, /* e.g., cuMemHostAlloc */

  GMM_MEM_TYPE_ALLOC = 3,

  GMM_MEM_TYPE_VMM = 4,
  GMM_MEM_TYPE_DEFAULT = 4,

  GMM_MEM_TYPE_GDR = 5, /* for small obj */
  GMM_MEM_TYPE_IPC = 6, /* for large obj */

};

// basic info for cuda dev mem allocation
struct CUDA_devMem {
  int dev_id;
  cuda_mem_type type;  // VMM, VMM+GDR, VMM+IPC, ...

  CUdeviceptr dptr;  // addr used by app (exclude GDR)

  size_t orig_size;   // orig alloc req size
  size_t alloc_size;  // actual alloc size due to alignment

  // for VMM
  CUmemGenericAllocationHandle vmm_handle;

  // for GDR
  gdr_mh_t mH;
  void *va_dptr;  // addr used in GDR
  void *map_dptr;
  gdr_info_t gdr_info;

 public:
  int get_devID() const { return dev_id; }
  CUdeviceptr get_addr() const { return dptr; }
  size_t get_alloc_size() const { return alloc_size; }
  size_t get_orig_size() const { return orig_size; }
  gdr_mh_t get_mHandle() const { return mH; }

  void *get_va_dptr() const { return va_dptr; }
  void *get_map_dptr() const { return map_dptr; }
  gdr_info_t get_gdr_info() const { return gdr_info; }
  cuda_mem_type get_type() const { return type; }
  CUmemGenericAllocationHandle get_vmm_handle() const { return vmm_handle; }

  gdr_info_t &get_gdr_info_ref() { return gdr_info; }
  gdr_mh_t &get_mHandle_ref() { return mH; }
  void *&get_map_dptr_ref() { return map_dptr; }
  void *&get_va_dptr_ref() { return va_dptr; }
  CUdeviceptr &get_addr_ref() { return dptr; }

  void set_type(cuda_mem_type t) { type = t; }

 public:
  CUDA_devMem(int dev_, CUdeviceptr dptr_, CUmemGenericAllocationHandle handle,
              size_t size_, size_t aligned_size_, cuda_mem_type type_) {
    dev_id = dev_, dptr = dptr_;
    vmm_handle = handle;
    orig_size = size_;
    alloc_size = aligned_size_;
    type = type_;
  }

  ~CUDA_devMem() {}
};

// alloc dev mem via VMM and export it
// assuming a mem pool thus limited dev mem allocation
CUresult gmm_cuda_vmm_alloc(size_t bytesize, int dev_id, CUdeviceptr &ret_ptr,
                            CUmemGenericAllocationHandle &vmm_handle,
                            size_t *alloc_size, bool enable_gdr,
                            bool rw = true);

CUresult gmm_cuda_vmm_free(CUdeviceptr &dptr, gmm_shmInfo_t *&shm);
/*
static const gdr_mh_t null_mh = {0};
static inline bool operator==(const gdr_mh_t &a, const gdr_mh_t &b) {
  return a.h == b.h;
}
*/
